#!/bin/bash
# Performance-lever sweep and production-image serving gates on the physical RTX 5090.
#
# Every phase runs the immutable production image (localhost/vllm:gg-r34-patched) with no
# source bind mounts: the three patch modules are baked in and re-verified inside the image
# before any GPU work, so a row measured here is a statement about the release artifact
# rather than about a mount.
#
# One invocation is one phase, so the sweep is interruptible at a phase boundary and the
# user's live service can be restored at any point:
#
#   perf_sweep_5090.sh stage             verify image + modules, freeze prompts (no GPU)
#   perf_sweep_5090.sh gates <recipe>    production-image serving gate for one card recipe
#   perf_sweep_5090.sh run <tag>         one perf configuration: start, measure, stop
#   perf_sweep_5090.sh fidelity <tag>    30-case image suite + longest needle that fits
#   perf_sweep_5090.sh restore           restore the user's live qwen38-27b service
#
# Deliberately NOT swept, with reasons recorded in receipts/perf-sweep-5090.json:
# B12X_ATTN (needs block size exactly 64/128; the hybrid mamba page forces >=1600 here),
# FULL_AND_PIECEWISE (refused by the exl3 graph-decode guard, plus Xid 31 history on SM120),
# VLLM_EXL3_PREFILL_FP8 (no-op: the pinned extension exports no fp8 reconstruct symbol),
# rejection_sample_method=synthetic (fabricates acceptance; never a serving option),
# --enable-prefix-caching (needs the -apc superset image, which is not the qualified one).
set -euo pipefail

W=${W:-/home/mbelleau/perfsweep}
OUT=$W/out
LOGS=$W/logs
PROMPTS=$W/prompts
TOOLS=$W/tools
PORT=${PORT:-8231}
HOST_LAN=${HOST_LAN:-10.15.0.151}
IMAGE=${IMAGE:-localhost/vllm:gg-r34-patched}
IMAGE_CONFIG_ID=${IMAGE_CONFIG_ID:-3cd11158a540ff4c23cece491774fecd3e61f6e4784771273e72bd65e44b773c}
IMAGE_MANIFEST=${IMAGE_MANIFEST:-sha256:6eca4c693f01b6f4e112c04eacd30673b7cfbba4150e6fe2ea3ba1bbfde14c27}
HF=${HF:-/mnt/vault/llm/huggingface}
CTX_REPO=$HF/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-context
CTX_REV=c45c273b0d6ef2859cb2d85b36dd52253c80d878
HYD_REPO=$HF/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated
HYD_REV=5112f59411a45547b5fc22c63fc35755175cf5f3
JIT=${JIT:-/home/mbelleau/qual5090/cache/jit}
ONLINE_CACHE=${ONLINE_CACHE:-/home/mbelleau/qual5090/cache/exl3-online}
VENV=/opt/venv/lib/python3.12/site-packages/vllm
EXL3_SHA=2df9d0799fd323798cead1edb773cab556c94798eec263ee03ded35408c6e4ee
QWEN_SHA=04d2bd587b37142f4f55a8d00b9f8c907309490168cb7fcdfde450531df2c9e7
MTP_SHA=0090dc131f0eaf439b24d50baf4def9f10b052864c76e695053d64f66b274bab
QUANT_CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
READY_TIMEOUT=${READY_TIMEOUT:-1200}
CONTAINER=
SAMPLER_PID=
LEVER=
declare -a podman_argv=() serve_argv=()

mkdir -p "$OUT" "$LOGS" "$PROMPTS" "$TOOLS"

die() { echo "FATAL: $*" >&2; exit 2; }
say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

cleanup() {
  local rc=$?
  if [[ -n $SAMPLER_PID ]]; then kill "$SAMPLER_PID" 2>/dev/null || true; fi
  if [[ -n $CONTAINER ]]; then
    podman stop -t 20 "$CONTAINER" >/dev/null 2>&1 || true
    podman rm -f "$CONTAINER" >/dev/null 2>&1 || true
  fi
  return $rc
}
trap cleanup EXIT INT TERM

verify_image() {
  local config_id manifest
  config_id=$(podman image inspect -f '{{.Id}}' "$IMAGE") || die "image $IMAGE missing"
  manifest=$(podman image inspect -f '{{index .RepoDigests 0}}' "$IMAGE" | sed 's/.*@//')
  [[ ${config_id#sha256:} == "${IMAGE_CONFIG_ID#sha256:}" ]] ||
    die "image config id $config_id != $IMAGE_CONFIG_ID"
  [[ $manifest == "$IMAGE_MANIFEST" ]] || die "image manifest $manifest != $IMAGE_MANIFEST"
  say "image identity verified: $IMAGE ($manifest)"
}

# The three modules must be the published bytes *inside the image*, resolved by the import
# machinery, with no bind mount, before any measurement. CPU-only container, no GPU device.
verify_modules_in_image() {
  podman run --rm --network none --entrypoint /bin/bash "$IMAGE" -lc "
    set -eu
    printf '%s  %s\n' \
      $EXL3_SHA $VENV/model_executor/layers/quantization/exl3.py \
      $QWEN_SHA $VENV/model_executor/models/qwen3_5.py \
      $MTP_SHA $VENV/model_executor/models/qwen3_5_mtp.py | sha256sum -c -
    /opt/venv/bin/python -c '
import hashlib, importlib.util, json
want = {\"vllm.model_executor.layers.quantization.exl3\": \"$EXL3_SHA\",
        \"vllm.model_executor.models.qwen3_5\": \"$QWEN_SHA\",
        \"vllm.model_executor.models.qwen3_5_mtp\": \"$MTP_SHA\"}
origins = {}
for name, digest in want.items():
    origin = importlib.util.find_spec(name).origin
    got = hashlib.sha256(open(origin, \"rb\").read()).hexdigest()
    assert got == digest, (name, origin, got, digest)
    origins[name] = origin
print(json.dumps({\"import_resolution_matches_published_bytes\": True, \"origins\": origins}))
'" | tee "$OUT/module-verify.txt"
}

start_sampler() {
  nvidia-smi --query-gpu=timestamp,memory.total,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader,nounits -lms 500 >"$LOGS/gpu-$1.csv" 2>/dev/null &
  SAMPLER_PID=$!
}

# Facts a reader needs to compare two rows: what the engine actually chose and allocated.
startup_facts() {
  local log=$1 tag=$2
  {
    grep -oE 'Using [A-Z_]+ attention backend out of potential backends: \[[^]]*\]' "$log" || true
    grep -oE 'Using AttentionBackendEnum\.[A-Z_]+ for MMEncoderAttention' "$log" || true
    grep -oE 'Setting attention block size to [0-9]+ tokens[^.]*.' "$log" || true
    grep -oE 'Padding mamba page size by [0-9.]+%[^.]*.' "$log" || true
    grep -oE 'Add [0-9]+ padding layers[^.]*' "$log" || true
    grep -oE 'Available KV cache memory: [0-9.]+ GiB' "$log" || true
    grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$log" || true
    grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' "$log" || true
    grep -oE 'Actual usage is .{0,200}' "$log" | head -1 || true
    grep -oE 'Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB' "$log" || true
    grep -oE 'Capturing CUDA graphs \(decode, FULL\): *100%[^0-9]*[0-9]+/[0-9]+' "$log" | tail -1 || true
    grep -oE 'Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds' "$log" || true
    grep -oE 'init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s[^)]*.' "$log" || true
    grep -oE 'enable_prefix_caching=[A-Za-z]+' "$log" | head -1 || true
    grep -oE 'num_spec_tokens=[0-9]+' "$log" | head -1 || true
    grep -oE 'cudagraph_capture_sizes=\[[^]]*\]' "$log" | head -1 || true
    grep -oE 'custom_ops=\[[^]]*\]' "$log" | head -1 || true
    grep -oE 'EXL3 graph decode enabled by [^.]*.' "$log" || true
    grep -oE 'priming exl3_gemm for [0-9]+ row counts \(m=[0-9]+\.\.[0-9]+\)' "$log" || true
    grep -oE 'Initializing a V1 LLM engine \(v[^)]*\)' "$log" | head -1 || true
    grep -oiE 'adaptive[a-z_]*|num_speculative_tokens_per_batch_size=[^,]*' "$log" | sort -u | head -5 || true
  } | sed 's/^ *//' | sort -u >"$OUT/startup-facts-$tag.txt"
  grep -F 'Initializing a V1 LLM engine' "$log" | head -1 >"$OUT/engine-config-$tag.txt" || true
  cat "$OUT/startup-facts-$tag.txt" >&2
}

# A configuration that refuses to start is a result, not a harness error: the caller records
# it and moves on, so one closed lever does not cost the rest of the window.
wait_ready() {
  local log=$1 i
  for ((i = 0; i < READY_TIMEOUT; i++)); do
    if curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      say "server ready after ${i}s"
      return 0
    fi
    podman container exists "$CONTAINER" 2>/dev/null || {
      say "container $CONTAINER exited before readiness"
      return 1
    }
    sleep 1
  done
  say "server not ready within ${READY_TIMEOUT}s"
  return 1
}

record_failure() {
  local tag=$1 log=$2
  python3 - "$OUT/failed-$tag.json" "$tag" "$LEVER" "$log" <<'PY'
import json, pathlib, re, sys
out, tag, lever, log = sys.argv[1:]
text = pathlib.Path(log).read_text(errors="ignore")
patterns = {
    "exl3_refusal": r"The EXL3 quantization backend requires eager execution[^\n]*",
    "cudagraph_override": r"Overriding cudagraph_mode from [^\n]*",
    "value_error": r"ValueError: [^\n]*",
    "engine_dead": r"EngineCore[^\n]*(failed|died|Traceback)[^\n]*",
}
found = {name: sorted(set(re.findall(pattern, text)))[:3] for name, pattern in patterns.items()}
pathlib.Path(out).write_text(json.dumps({
    "schema": "qwen38-perf-sweep-failure/1", "tag": tag, "lever": lever,
    "started": False, "outcome": "server refused to start on this profile",
    "evidence": {name: value for name, value in found.items() if value},
    "log_tail": text.splitlines()[-25:]}, indent=2) + "\n")
print(json.dumps({"tag": tag, "started": False,
                  "evidence": [v[0] for v in found.values() if v][:2]}))
PY
}

compose_podman() {
  CONTAINER=$1
  local repo=$2 mount=$3
  shift 3
  podman_argv=(
    podman run --rm --name "$CONTAINER"
    --device nvidia.com/gpu=all --ipc=host --network host
    -v "$repo:$mount:ro"
    -v "$JIT:/cache/jit:rw"
    -v "$ONLINE_CACHE:/cache/exl3-online:rw"
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "$@"
    --entrypoint /opt/venv/bin/vllm
    "$IMAGE"
  )
}

write_command_json() {
  local tag=$1
  python3 - "$OUT/command-$tag.json" "$tag" "$LEVER" "$IMAGE" "$IMAGE_MANIFEST" \
    "${podman_argv[@]}" -- "${serve_argv[@]}" <<'PY'
import json, pathlib, sys
out, tag, lever, image, manifest, *rest = sys.argv[1:]
split = rest.index("--")
pathlib.Path(out).write_text(json.dumps({
    "schema": "qwen38-perf-sweep-command/1", "tag": tag, "lever": lever,
    "image": image, "image_manifest_digest": manifest,
    "source_bind_mounts": [],
    "podman_argv": rest[:split], "vllm_argv": rest[split + 1:]},
    indent=2) + "\n")
PY
}

launch() {
  local tag=$1 log=$2
  say "launching $tag"
  "${podman_argv[@]}" "${serve_argv[@]}" >"$log" 2>&1 &
  wait_ready "$log"
}

stop_server() {
  podman stop -t 20 "$CONTAINER" >/dev/null 2>&1 || true
  podman rm -f "$CONTAINER" >/dev/null 2>&1 || true
  CONTAINER=
  if [[ -n $SAMPLER_PID ]]; then kill "$SAMPLER_PID" 2>/dev/null || true; SAMPLER_PID=; fi
  local i
  for ((i = 0; i < 90; i++)); do
    [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 500 ]] && return 0
    sleep 1
  done
  say "warning: GPU memory still held after teardown"
}

peak_mib() { awk -F', ' '$3+0 > m { m = $3+0 } END { print m+0 }' "$LOGS/gpu-$1.csv"; }
kv_tokens() {
  grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$OUT/startup-facts-$1.txt" |
    grep -oE '[0-9,]+' | tr -d , | head -1
}

# ---------------------------------------------------------------- perf configurations
# Each configuration changes exactly one thing against `base`. cudagraph_capture_sizes lists
# only the largest shape: the engine expands it to every request count from 1 to max_num_seqs
# (CompilationConfig.adjust_cudagraph_sizes_for_spec_decode adds (1+num_spec)*num_reqs for
# num_reqs 1..32), so C1/C4/C8 each replay a captured decode graph instead of falling back.
SEQS=8
SPEC='{"method":"mtp","num_speculative_tokens":3}'
COMPILE='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[32]}'
MM_CFG='{"truncation":false,"max_pixels":8388608}'
declare -a EXTRA_ARGS=() EXTRA_ENV=()

configure() {
  case $1 in
  base) LEVER='none (reference row for every A/B)' ;;
  customops)
    LEVER='L7 custom_ops:["all"] (the vendor reference launcher setting; not bit-exact)'
    COMPILE='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[32],"custom_ops":["all"]}'
    ;;
  flashinfer)
    LEVER='L2 --attention-backend FLASHINFER explicit, attention workspace inside the memory profile'
    EXTRA_ARGS=(--attention-backend FLASHINFER)
    EXTRA_ENV=(-e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1)
    ;;
  triton)
    LEVER='L2 control --attention-backend TRITON_ATTN (what the live K4 service forces)'
    EXTRA_ARGS=(--attention-backend TRITON_ATTN)
    ;;
  mtp1)
    LEVER='L6 speculative depth 1'
    SPEC='{"method":"mtp","num_speculative_tokens":1}'
    COMPILE='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[16]}'
    ;;
  mtp2)
    LEVER='L6 speculative depth 2'
    SPEC='{"method":"mtp","num_speculative_tokens":2}'
    COMPILE='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[24]}'
    ;;
  adaptive)
    LEVER='L6 adaptive_speculative_tokens_window=8, depth 3 as the initial value and upper bound'
    SPEC='{"method":"mtp","num_speculative_tokens":3,"adaptive_speculative_tokens_window":8}'
    ;;
  perbatch)
    LEVER='L6 num_speculative_tokens_per_batch_size: depth 3 at batch 1-2, depth 1 at batch 3-8'
    SPEC='{"method":"mtp","num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,2,3],[3,8,1]]}'
    ;;
  # Dynamic speculative decoding (either knob) makes VllmConfig
  # _maybe_override_dynamic_sd_cudagraph_mode downgrade cudagraph_mode to PIECEWISE on the
  # MRV1 runner, and PIECEWISE has a non-NONE mixed mode, which Exl3Config refuses: the
  # server then raises "requires eager execution" at load. So each dynamic lever is measured
  # twice: once as-is on the qualified profile (expected: startup refusal, recorded as the
  # result) and once with --enforce-eager, against `eagerbase` rather than `base`, so the
  # dynamic effect is not confused with the cost of losing graph decode.
  eagerbase)
    LEVER='reference for the eager arms: depth 3, --enforce-eager, no CUDA-graph decode'
    COMPILE='{"cudagraph_mode":"NONE"}'
    EXTRA_ARGS=(--enforce-eager)
    ;;
  adaptive_eager)
    LEVER='L6 adaptive_speculative_tokens_window=8 with --enforce-eager (the only way this build runs it)'
    SPEC='{"method":"mtp","num_speculative_tokens":3,"adaptive_speculative_tokens_window":8}'
    COMPILE='{"cudagraph_mode":"NONE"}'
    EXTRA_ARGS=(--enforce-eager)
    ;;
  perbatch_eager)
    LEVER='L6 num_speculative_tokens_per_batch_size (3 at batch 1-2, 1 at batch 3-8) with --enforce-eager'
    SPEC='{"method":"mtp","num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,2,3],[3,8,1]]}'
    COMPILE='{"cudagraph_mode":"NONE"}'
    EXTRA_ARGS=(--enforce-eager)
    ;;
  combo)
    LEVER="composed winner: ${COMBO_LEVER:?set COMBO_LEVER for the combo tag}"
    SPEC=${COMBO_SPEC:-$SPEC}
    COMPILE=${COMBO_COMPILE:-$COMPILE}
    [[ -n ${COMBO_ARGS:-} ]] && read -r -a EXTRA_ARGS <<<"$COMBO_ARGS"
    [[ -n ${COMBO_ENV:-} ]] && read -r -a EXTRA_ENV <<<"$COMBO_ENV"
    ;;
  *) die "unknown configuration '$1'" ;;
  esac
}

perf_serve_argv() {
  serve_argv=(
    serve "/models/ctx-repo/snapshots/$CTX_REV"
    --served-model-name m
    --quantization exl3 --quantization-config "$QUANT_CFG"
    --max-model-len 262144
    --gpu-memory-utilization 0.97
    --kv-cache-dtype fp8
    --max-num-seqs "$SEQS"
    --max-num-batched-tokens 2048
    --speculative-config "$SPEC"
    --mm-processor-kwargs "$MM_CFG"
    --compilation-config "$COMPILE"
    "${EXTRA_ARGS[@]}"
    --host 127.0.0.1 --port "$PORT"
  )
}

# ---------------------------------------------------------------- phases
phase_stage() {
  verify_image
  verify_modules_in_image
  [[ -d $CTX_REPO/snapshots/$CTX_REV ]] || die "context snapshot missing at $CTX_REPO"
  mkdir -p "$W/corpus"
  [[ -d $W/corpus/literary ]] ||
    cp -r "${CORPUS_SRC:-/home/mbelleau/qual5090/corpus/literary}" "$W/corpus/"
  if [[ ! -f $PROMPTS/manifest.json ]]; then
    podman run --rm --network none \
      -v "$CTX_REPO:/models/ctx-repo:ro" -v "$W:/work:rw" \
      --entrypoint /opt/venv/bin/python "$IMAGE" \
      /work/tools/perf_prompts_5090.py \
      --model "/models/ctx-repo/snapshots/$CTX_REV" \
      --corpus /work/corpus/literary --out-dir /work/prompts
  fi
  python3 - "$OUT/stage.json" "$W" "$IMAGE" "$IMAGE_MANIFEST" "$IMAGE_CONFIG_ID" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out, work, image, manifest, config_id = sys.argv[1:]
work = pathlib.Path(work)
def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema": "qwen38-perf-sweep-stage/1",
    "host": run("hostname"),
    "gpu": run("nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version",
               "--format=csv,noheader"),
    "podman_version": run("podman", "--version"),
    "image": image, "image_manifest_digest": manifest, "image_config_id": config_id,
    "patch_modules_verified_inside_image": True,
    "source_bind_mounts_used": [],
    "harness_sha256": {p.name: sha(p) for p in sorted(work.glob("tools/*")) if p.is_file()},
    "prompt_manifest": json.loads((work / "prompts" / "manifest.json").read_text()),
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"stage": out, "prompts": len(payload["prompt_manifest"]["prompts"])}))
PY
  say "stage complete"
}

phase_run() {
  local tag=$1 log=$LOGS/server-$tag.log
  configure "$tag"
  compose_podman "perfsweep-$tag" "$CTX_REPO" /models/ctx-repo \
    -e VLLM_EXL3_EMBED_BITS=8 -e VLLM_EXL3_GRAPH_DECODE=1 \
    -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 "${EXTRA_ENV[@]}"
  perf_serve_argv
  write_command_json "$tag"
  start_sampler "$tag"
  if ! launch "$tag" "$log"; then
    startup_facts "$log" "$tag"
    record_failure "$tag" "$log"
    stop_server
    say "run $tag recorded as a startup refusal"
    return 0
  fi
  startup_facts "$log" "$tag"
  exposure_probe "$tag"
  # shellcheck disable=SC2086
  python3 "$TOOLS/perf_probe_5090.py" --base "http://127.0.0.1:$PORT" --label "$tag" \
    --prompt-dir "$PROMPTS" --out "$OUT/probe-$tag.json" ${PROBE_ARGS:-} \
    >"$LOGS/probe-$tag.out" 2>"$LOGS/probe-$tag.err" ||
    { tail -20 "$LOGS/probe-$tag.err" >&2; die "probe failed for $tag"; }
  cat "$LOGS/probe-$tag.out" >&2
  python3 - "$OUT/memory-$tag.json" "$tag" "$(peak_mib "$tag")" "$(kv_tokens "$tag")" <<'PY'
import json, pathlib, sys
out, tag, peak, kv = sys.argv[1:]
pathlib.Path(out).write_text(json.dumps(
    {"tag": tag, "peak_gpu_memory_used_mib": int(peak or 0),
     "gpu_kv_cache_tokens": int(kv or 0),
     "sampler": "nvidia-smi memory.used every 500 ms for the whole phase"},
    indent=2) + "\n")
print(json.dumps({"tag": tag, "peak_mib": int(peak or 0), "kv_tokens": int(kv or 0)}))
PY
  stop_server
  say "run $tag complete"
}

phase_fidelity() {
  local tag=$1 log=$LOGS/server-fidelity-$tag.log api=http://127.0.0.1:$PORT/v1 kv needle
  configure "$tag"
  compose_podman "perfsweep-fidelity-$tag" "$CTX_REPO" /models/ctx-repo \
    -e VLLM_EXL3_EMBED_BITS=8 -e VLLM_EXL3_GRAPH_DECODE=1 \
    -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 "${EXTRA_ENV[@]}"
  perf_serve_argv
  write_command_json "fidelity-$tag"
  start_sampler "fidelity-$tag"
  if ! launch "fidelity-$tag" "$log"; then
    record_failure "fidelity-$tag" "$log"
    stop_server
    die "fidelity server for $tag refused to start"
  fi
  startup_facts "$log" "fidelity-$tag"
  python3 "$TOOLS/vision_eval.py" --url "$api" --label "perfsweep-$tag" --cases 10 \
    --out "$OUT/fidelity-vision-$tag.json" >"$LOGS/fidelity-vision-$tag.out" 2>&1
  tail -1 "$LOGS/fidelity-vision-$tag.out" >&2
  kv=$(kv_tokens "fidelity-$tag")
  # Longest needle that fits: on this card the KV pool, not max_model_len, is binding once
  # the concurrency profile's graph pool is paid for.
  needle=$((kv * 95 / 100))
  ((needle > 261800)) && needle=261800
  python3 "$TOOLS/longctx.py" --url "$api" --corpus "$W/corpus/literary" \
    --tokens "$needle" --depth 0.5 --out "$OUT/fidelity-needle-$tag.json" --require-pass \
    >"$LOGS/fidelity-needle-$tag.out" 2>&1
  tail -2 "$LOGS/fidelity-needle-$tag.out" >&2
  stop_server
  say "fidelity $tag complete (needle at $needle tokens)"
}

# The production-image serving gates (four card recipes, smoke, read-only weights, podman
# diff, rootless uid) are owned by docker/build-image.sh in ImmutableImage's tree, which
# writes receipts/production-image.json itself. They are deliberately NOT reimplemented here:
# one owner per fact. What is measured here instead is this sweep's own exposure model, which
# differs from that harness's --publish 127.0.0.1:8137:8000: these servers run --network host
# with --host 127.0.0.1, so loopback-only has to be demonstrated rather than assumed.
exposure_probe() {
  local tag=$1 loopback lan
  loopback=$(curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 &&
    echo true || echo false)
  lan=$(curl -sf --max-time 5 "http://$HOST_LAN:$PORT/health" >/dev/null 2>&1 &&
    echo true || echo false)
  python3 - "$OUT/exposure-$tag.json" "$tag" "$PORT" "$HOST_LAN" "$loopback" "$lan" <<'PY'
import json, pathlib, subprocess, sys
out, tag, port, lan_addr, loopback, lan = sys.argv[1:]
pathlib.Path(out).write_text(json.dumps({
    "schema": "qwen38-perf-sweep-exposure/1", "tag": tag, "port": int(port),
    "network_mode": "podman --network host, vllm --host 127.0.0.1",
    "reachable_on_loopback": loopback == "true",
    "lan_address_probed": lan_addr,
    "reachable_on_lan_address": lan == "true",
    "loopback_only": loopback == "true" and lan != "true",
    "api_key_required": False,
    "host_uid": subprocess.run(["id", "-u"], capture_output=True, text=True).stdout.strip(),
}, indent=2) + "\n")
print(json.dumps({"loopback_only": loopback == "true" and lan != "true"}))
PY
}

phase_restore() {
  say "restoring the user's live service"
  systemctl --user start qwen38-27b.service
  local i status health reachable
  for ((i = 0; i < 1200; i++)); do
    health=$(podman inspect -f '{{.State.Health.Status}}' qwen38-27b 2>/dev/null || echo none)
    [[ $health == healthy ]] && break
    sleep 2
  done
  status=$(systemctl --user is-active qwen38-27b.service || true)
  if curl -sf --max-time 10 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    reachable=true
  else
    reachable=false
  fi
  python3 - "$OUT/restore.json" "$status" "$health" "$reachable" <<'PY'
import json, pathlib, subprocess, sys
out, status, health, reachable = sys.argv[1:]
def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
pathlib.Path(out).write_text(json.dumps({
    "schema": "qwen38-perf-sweep-restore/1", "unit_active": status,
    "container_health": health, "health_endpoint_ok": reachable == "true",
    "image": run("podman", "inspect", "qwen38-27b", "--format", "{{.ImageName}}"),
    "binds": run("podman", "inspect", "qwen38-27b", "--format", "{{json .HostConfig.Binds}}"),
    "state": run("podman", "inspect", "qwen38-27b", "--format", "{{.State.Status}}"),
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
run) phase_run "${2:?tag required}" ;;
fidelity) phase_fidelity "${2:?tag required}" ;;
restore) phase_restore ;;
*) die "usage: $0 {stage|run <tag>|fidelity <tag>|restore}" ;;
esac
