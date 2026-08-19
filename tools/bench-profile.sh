#!/bin/bash
# Phase-0 multi-boot benchmark for the Qwen3.8-27B EXL3 serving stack.
#
# Reboots the qwen38-27b service N times with a given env override set, runs
# the canonical PP / TG-fox / TG-essay / acceptance / context-length probes on
# each boot, and emits a JSON receipt with per-boot raw results plus an
# aggregate.  The harness is designed to resolve 3% effects: it uses
# median-of-N across boots (not median-of-5 on a single boot), records wall
# times and GPU telemetry per boot, and records dead-boot failures without
# aborting the remaining boots.
#
# On exit — success or failure — the service is always restored to the DEFAULT
# flagship config (plain `systemctl --user start`, no env overrides) via a bash
# trap.
#
# Usage:
#   bench-profile.sh --name <profile> [--boots 3] [--out <json>] [--env KEY=VAL]...
#
# Each --env KEY=VAL is passed to the launcher for every boot.  Example:
#   bench-profile.sh --name fp8dg-m256 --boots 5 \
#       --env VLLM_EXL3_FP8DG_PREFILL_M=256 --env VLLM_EXL3_FP8DG_CACHE=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$SCRIPT_DIR"
REPO_ROOT="/home/mbelleau/qwen38-27b-exl3"
LAUNCHER="/home/mbelleau/run-qwen38-27b.sh"
SERVICE="qwen38-27b.service"
CONTAINER="qwen38-27b"

# --------------------------------------------------------------------------- usage
usage() {
    cat <<'EOF'
Usage: bench-profile.sh --name <profile> [--boots 3] [--out <json>] [--env KEY=VAL]...

Required:
  --name <profile>      Name tag for this benchmark profile (used in output path)

Optional:
  --boots <N>           Number of boot/measure cycles (default: 3)
  --out <path>          Output JSON receipt path
                        (default: receipts/bench-<name>-<UTC-date>.json)
  --env KEY=VAL         Environment override passed to the launcher for every boot.
                        May be repeated: --env MAX_MODEL_LEN=8192 --env MTP=6

The script does NOT touch systemd/podman in --help mode.

On completion (success or failure), the service is restored to the default
flagship config via `systemctl --user start qwen38-27b.service` (no env
overrides).
EOF
}

# --------------------------------------------------------------------------- parse args
NAME=""
BOOTS=3
OUT=""
declare -a ENV_VARS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)  NAME="$2"; shift 2 ;;
        --boots) BOOTS="$2"; shift 2 ;;
        --out)   OUT="$2"; shift 2 ;;
        --env)   ENV_VARS+=("$2"); shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$NAME" ]]; then
    echo "ERROR: --name is required" >&2
    usage >&2
    exit 1
fi

if ! [[ "$BOOTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --boots must be a positive integer, got: $BOOTS" >&2
    exit 1
fi

if [[ -z "$OUT" ]]; then
    UTC_DATE="$(date -u +%Y-%m-%d)"
    OUT="${REPO_ROOT}/receipts/bench-${NAME}-${UTC_DATE}.json"
fi

mkdir -p "$(dirname "$OUT")"

# --------------------------------------------------------------------------- helpers
say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

# stop_service: stop systemd unit and remove the container so the next boot
# starts from a clean slate.
stop_service() {
    say "Stopping ${SERVICE}..."
    systemctl --user stop "$SERVICE" 2>/dev/null || true
    sleep 2
    podman rm -f "$CONTAINER" 2>/dev/null || true
}

# start_with_env: launch the service with the collected env overrides.
# We call the launcher directly (not systemctl start) so env overrides take
# effect.  The launcher runs podman in detached mode (FOREGROUND is not set).
start_with_env() {
    say "Launching with env overrides: ${ENV_VARS[*]:-(none)}"
    if [[ ${#ENV_VARS[@]} -gt 0 ]]; then
        env "${ENV_VARS[@]}" "$LAUNCHER"
    else
        "$LAUNCHER"
    fi
}

# restore_default: start the service with the default flagship config (no env
# overrides) via systemd.  This is the EXIT trap target.
restore_default() {
    say "Restoring default flagship config..."
    stop_service
    say "Starting ${SERVICE} with default config..."
    systemctl --user start "$SERVICE" || true
    say "Default config restore initiated."
}

# --------------------------------------------------------------------------- EXIT trap
# CRITICAL: always leave the service running with the default flagship config,
# whether the benchmark succeeded, failed, or was interrupted.
trap restore_default EXIT

# --------------------------------------------------------------------------- temp files for results
RESULTS_FILE="$(mktemp /tmp/bench-profile-results.XXXXXX.jsonl)"
# Clean up the temp file after the restore_default trap runs.  We append the
# rm to the EXIT trap rather than replacing it: the service restore MUST run
# first (it is the safety-critical guarantee), then the temp file is removed.
trap 'restore_default; rm -f "$RESULTS_FILE"' EXIT

# Serialize env vars as JSON for the Python receipt builder.
ENV_VARS_JSON="[]"
if [[ ${#ENV_VARS[@]} -gt 0 ]]; then
    ENV_VARS_JSON=$(printf '%s\n' "${ENV_VARS[@]}" | python3 -c "
import sys, json
items = [line.strip() for line in sys.stdin if line.strip()]
print(json.dumps(items))
")
fi

# --------------------------------------------------------------------------- per-boot measurement
# run_one_boot: stop, launch with env, wait healthy, measure, append JSONL.
# Sets global BOOT_DIED.
BOOT_DIED=0

run_one_boot() {
    local boot_idx="$1"
    BOOT_DIED=0

    stop_service
    say "Boot ${boot_idx}/${BOOTS}: launching..."

    # Time the launcher itself. bench_lib.wait_healthy() cannot measure boot:
    # run-qwen38-27b.sh already blocks until /health is green before it
    # returns, so wait_healthy() afterwards always saw an up server and
    # reported ~0.0s (the "Boot time: 0.0+/-0.0s" bug).
    local launch_t0 launch_rc=0
    launch_t0=$(date +%s.%N)
    start_with_env || launch_rc=$?
    BOOT_WALL_SECONDS=$(awk -v a="$launch_t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')

    if [[ $launch_rc -ne 0 ]]; then
        say "Boot ${boot_idx}: launcher exited ${launch_rc}"
        python3 -c "
import json
print(json.dumps({'boot_index': ${boot_idx}, 'status': 'launch_failed',
                   'launcher_exit_code': ${launch_rc}}))
" >> "$RESULTS_FILE"
        BOOT_DIED=1
        return
    fi

    # Run all measurements in a single Python process.  stdout = JSON result
    # (one line); stderr = warnings/diagnostics.
    python3 -c "
import sys, json, time
sys.path.insert(0, '${TOOLS_DIR}')
import bench_lib

boot_idx = ${boot_idx}
result = {'boot_index': boot_idx, 'status': 'ok'}

try:
    # Confirm health (cheap here, the launcher already waited) and report the
    # launcher's own wall time as boot_seconds.
    bench_lib.wait_healthy(timeout_s=600)
    result['boot_seconds'] = float('${BOOT_WALL_SECONDS:-0}')
    result['gpu_telemetry_boot'] = bench_lib.gpu_telemetry()

    try:
        result['sanity'] = bench_lib.sanity()
    except Exception as e:
        result['sanity'] = None
        result['sanity_error'] = str(e)

    try:
        result['pp'] = bench_lib.measure_pp(reps=5)
    except Exception as e:
        result['pp'] = None
        result['pp_error'] = str(e)

    try:
        result['tg_fox'] = bench_lib.measure_tg(
            bench_lib.TG_FOX_PROMPT, max_tokens=200, reps=3)
    except Exception as e:
        result['tg_fox'] = None
        result['tg_fox_error'] = str(e)

    try:
        result['tg_essay'] = bench_lib.measure_tg(
            bench_lib.TG_ESSAY_PROMPT, max_tokens=500, reps=3)
    except Exception as e:
        result['tg_essay'] = None
        result['tg_essay_error'] = str(e)

    if result.get('tg_essay') and result['tg_essay'].get('acceptance') is not None:
        result['acceptance_essay'] = result['tg_essay']['acceptance']
    else:
        result['acceptance_essay'] = None

    # The objective requires MTP acceptance alongside EVERY TG number, and
    # acceptance is prompt-dependent: the fox prompt is highly predictable
    # (~0.93 accepted) while the essay prompt is not (~0.28-0.30).  Reporting
    # only the essay figure next to a fox throughput number is misleading.
    if result.get('tg_fox') and result['tg_fox'].get('acceptance') is not None:
        result['acceptance_fox'] = result['tg_fox']['acceptance']
    else:
        result['acceptance_fox'] = None

    try:
        result['max_model_len'] = bench_lib.max_model_len()
    except Exception as e:
        result['max_model_len'] = None
        result['max_model_len_error'] = str(e)

    result['gpu_telemetry_after'] = bench_lib.gpu_telemetry()

except bench_lib.BootDied as e:
    result['status'] = 'boot_died'
    result['error'] = str(e)
    result['error_lines'] = e.error_lines
except TimeoutError as e:
    result['status'] = 'timeout'
    result['error'] = str(e)
except Exception as e:
    result['status'] = 'error'
    result['error'] = str(e)

print(json.dumps(result, default=str), flush=True)
" >> "$RESULTS_FILE" || {
        # If the Python process itself failed, record a measurement error.
        python3 -c "
import json
print(json.dumps({'boot_index': ${boot_idx}, 'status': 'measurement_error',
                   'error': 'python measurement process failed'}))
" >> "$RESULTS_FILE"
        BOOT_DIED=1
        return
    }

    # Check if the boot died based on the status field.
    local status
    status=$(tail -1 "$RESULTS_FILE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','ok'))" 2>/dev/null || echo "error")
    if [[ "$status" != "ok" ]]; then
        BOOT_DIED=1
    fi
}

# --------------------------------------------------------------------------- main loop
say "bench-profile: name=${NAME} boots=${BOOTS} out=${OUT}"
say "Env overrides: ${ENV_VARS[*]:-(none)}"

ANY_DIED=0

for i in $(seq 1 "$BOOTS"); do
    run_one_boot "$i"
    if [[ "$BOOT_DIED" -eq 1 ]]; then
        ANY_DIED=1
        say "Boot ${i}: DIED/FAILED"
    else
        say "Boot ${i}: OK"
    fi
done

# --------------------------------------------------------------------------- aggregate + emit JSON
say "Aggregating results..."

UTC_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

python3 -c "
import sys, json
sys.path.insert(0, '${TOOLS_DIR}')
import bench_lib

# Read JSONL results file.
results = []
with open('${RESULTS_FILE}') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            pass

# Collect metric lists across successful boots.
pp_tok_s = []
tg_fox_tok_s = []
tg_essay_tok_s = []
acceptance_essay = []
acceptance_fox = []
boot_seconds = []
clocks_sm = []
power_draw = []
temp_gpu = []

for r in results:
    if r.get('status') != 'ok':
        continue
    if r.get('pp'):
        pp_tok_s.append(r['pp']['tok_s_median'])
    if r.get('tg_fox'):
        tg_fox_tok_s.append(r['tg_fox']['tok_s_median'])
    if r.get('tg_essay'):
        tg_essay_tok_s.append(r['tg_essay']['tok_s_median'])
        acc = r['tg_essay'].get('acceptance')
        if acc is not None:
            acceptance_essay.append(acc)
    if r.get('tg_fox'):
        accf = r['tg_fox'].get('acceptance')
        if accf is not None:
            acceptance_fox.append(accf)
    if r.get('boot_seconds') is not None:
        boot_seconds.append(r['boot_seconds'])
    for telo_key in ('gpu_telemetry_boot', 'gpu_telemetry_after'):
        telo = r.get(telo_key, {})
        if telo.get('clocks_sm', 0) > 0:
            clocks_sm.append(telo['clocks_sm'])
        if telo.get('power_draw', 0) > 0:
            power_draw.append(telo['power_draw'])
        if telo.get('temperature_gpu', 0) > 0:
            temp_gpu.append(telo['temperature_gpu'])

def safe_stats(vals):
    return bench_lib.stats(vals) if vals else {'mean': 0.0, 'sd': 0.0, 'n': 0, 'min': 0.0, 'max': 0.0}

receipt = {
    'schema': bench_lib.SCHEMA,
    'name': '${NAME}',
    'boots': ${BOOTS},
    'env': ${ENV_VARS_JSON},
    'utc_date': '${UTC_TS}',
    'boots_raw': results,
    'aggregate': {
        'pp_tok_s': safe_stats(pp_tok_s),
        'tg_fox_tok_s': safe_stats(tg_fox_tok_s),
        'tg_essay_tok_s': safe_stats(tg_essay_tok_s),
        'acceptance_essay': safe_stats(acceptance_essay),
        'acceptance_fox': safe_stats(acceptance_fox),
        'boot_seconds': safe_stats(boot_seconds),
        'telemetry_range': {
            'clocks_sm': safe_stats(clocks_sm),
            'power_draw': safe_stats(power_draw),
            'temperature_gpu': safe_stats(temp_gpu),
        },
    },
}
print(json.dumps(receipt, indent=2, default=str))
" > "$OUT"

say "JSON receipt written to: $OUT"

# --------------------------------------------------------------------------- human summary table
echo ""
echo "=== Benchmark Summary: ${NAME} ==="
echo "Boots: ${BOOTS} | Output: ${OUT}"
echo ""

python3 -c "
import sys, json
sys.path.insert(0, '${TOOLS_DIR}')
import bench_lib

with open('${OUT}') as f:
    receipt = json.load(f)
agg = receipt.get('aggregate', {})

def fmt(key, unit=''):
    s = agg.get(key, {})
    if s.get('n', 0) == 0:
        return 'N/A'
    return f\"{s['mean']:.1f}+/-{s['sd']:.1f} {unit} (n={s['n']})\"

def accfmt(key):
    a = agg.get(key, {})
    if a.get('n', 0) > 0:
        return f\"  [MTP acc {a['mean']:.3f}+/-{a['sd']:.3f}]\"
    return '  [MTP acc N/A]'


print(f\"  PP:          {fmt('pp_tok_s', 'tok/s')}\")
print(f\"  TG-fox:      {fmt('tg_fox_tok_s', 'tok/s')}\"
      f\"{accfmt('acceptance_fox')}\")
print(f\"  TG-essay:    {fmt('tg_essay_tok_s', 'tok/s')}\"
      f\"{accfmt('acceptance_essay')}\")

bs = agg.get('boot_seconds', {})
if bs.get('n', 0) > 0:
    print(f\"  Boot time:   {bs['mean']:.1f}+/-{bs['sd']:.1f}s (n={bs['n']})\")

dead = [b for b in receipt.get('boots_raw', []) if b.get('status') != 'ok']
if dead:
    print(f\"\n  DEAD/FAILED boots: {len(dead)}\")
    for b in dead:
        print(f\"    boot {b.get('boot_index','?')}: {b.get('status','?')} - {b.get('error','')}\")
"

echo ""

# Exit 2 if any boot died.
if [[ "$ANY_DIED" -eq 1 ]]; then
    say "One or more boots died - exiting with code 2."
    exit 2
fi

say "All boots completed successfully."
exit 0
